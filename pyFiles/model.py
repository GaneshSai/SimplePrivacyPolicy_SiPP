'''

WRITE THE FUNCTION METHOD HERE. 

Input -> Text 

return Json in format: 

Output:
{
    "policy"  : {
        "Data Retention": ["Data Retention - 1", "Data Retention -  2"],
        "First Party Collection/Use": ["First Party Collection/Use - 1", "First Party Collection/Use -  2"],
        "International and Specific Audiences": ["International and Specific Audiences - 1", "International and Specific Audiences -  2"],
        "Other": ["Other - 1", "Other -  2"],
        "Policy Change": ["Policy Change - 1", "Policy Change -  2"],
        "Third Party Sharing/Collection": ["Third Party Sharing/Collection - 1", "Third Party Sharing/Collection -  2"],
        "User Access, Edit and Deletion": ["User Access, Edit and Deletion - 1", "User Access, Edit and Deletion -  2"],
        "User Choice/Control": ["User Choice/Control - 1", "User Choice/Control -  2"],
        "Data Security": ["Data Security - 1", "Data Security -  2"],
        "Do Not Track": ["Do Not Track - 1", "Do Not Track -  2"]
    }
}

'''

def SiPP(policy_text): 
    output_json = {
        "policy"  : {
            "Data Retention": ["Data Retention - 1", "Data Retention -  2"],
            "First Party Collection/Use": ["First Party Collection/Use - 1", "First Party Collection/Use -  2"],
            "International and Specific Audiences": ["International and Specific Audiences - 1", "International and Specific Audiences -  2"],
            "Other": ["Other - 1", "Other -  2"],
            "Policy Change": ["Policy Change - 1", "Policy Change -  2"],
            "Third Party Sharing/Collection": ["Third Party Sharing/Collection - 1", "Third Party Sharing/Collection -  2"],
            "User Access, Edit and Deletion": ["User Access, Edit and Deletion - 1", "User Access, Edit and Deletion -  2"],
            "User Choice/Control": ["User Choice/Control - 1", "User Choice/Control -  2"],
            "Data Security": ["Data Security - 1", "Data Security -  2"],
            "Do Not Track": ["Do Not Track - 1", "Do Not Track -  2"]
        }
    }
    return output_json